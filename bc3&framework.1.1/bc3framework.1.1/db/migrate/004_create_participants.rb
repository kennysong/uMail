class CreateParticipants < ActiveRecord::Migration
  def self.up
    create_table :participants do |t|
      # t.column :name, :string
      t.column :id, :primary_key
      t.column :Name, :string
      t.column :Email, :string
      t.column :Class, :string
      t.column :Department, :string
      t.column :Again, :boolean
      t.column :Recruitment, :string
    end
  end

  def self.down
    drop_table :participants
  end
end
